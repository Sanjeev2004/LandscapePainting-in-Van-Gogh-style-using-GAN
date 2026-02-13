import argparse
import pathlib
from typing import Tuple

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def normalize(image: tf.Tensor) -> tf.Tensor:
    return (tf.cast(image, tf.float32) / 127.5) - 1.0


def denormalize(image: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value((image + 1.0) * 127.5, 0.0, 255.0)


def decode_and_resize(path: tf.Tensor, image_size: int, training: bool) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    if training:
        image = tf.image.resize(image, [image_size + 30, image_size + 30])
        image = tf.image.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
    else:
        image = tf.image.resize(image, [image_size, image_size])
    return normalize(image)


def make_dataset(folder: pathlib.Path, image_size: int, batch_size: int, training: bool) -> tf.data.Dataset:
    pattern = str(folder / "*")
    paths = tf.data.Dataset.list_files(pattern, shuffle=training)
    ds = paths.map(lambda p: decode_and_resize(p, image_size, training), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(1024)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def downsample(filters: int, size: int, apply_norm: bool = True) -> tf.keras.Sequential:
    init = tf.random_normal_initializer(0.0, 0.02)
    layers = [
        tf.keras.layers.Conv2D(filters, size, strides=2, padding="same", kernel_initializer=init, use_bias=False),
    ]
    if apply_norm:
        layers.append(tf.keras.layers.LayerNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    return tf.keras.Sequential(layers)


def upsample(filters: int, size: int, apply_dropout: bool = False) -> tf.keras.Sequential:
    init = tf.random_normal_initializer(0.0, 0.02)
    layers = [
        tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2, padding="same", kernel_initializer=init, use_bias=False
        ),
        tf.keras.layers.LayerNormalization(),
    ]
    if apply_dropout:
        layers.append(tf.keras.layers.Dropout(0.5))
    layers.append(tf.keras.layers.ReLU())
    return tf.keras.Sequential(layers)


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def call(self, x: tf.Tensor) -> tf.Tensor:
        pad_h, pad_w = self.padding
        return tf.pad(x, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="REFLECT")


def residual_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    init = tf.random_normal_initializer(0.0, 0.02)
    y = ReflectionPadding2D((1, 1))(x)
    y = tf.keras.layers.Conv2D(filters, 3, padding="valid", kernel_initializer=init, use_bias=False)(y)
    y = tf.keras.layers.LayerNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = ReflectionPadding2D((1, 1))(y)
    y = tf.keras.layers.Conv2D(filters, 3, padding="valid", kernel_initializer=init, use_bias=False)(y)
    y = tf.keras.layers.LayerNormalization()(y)
    return tf.keras.layers.Add()([x, y])


def build_resnet_generator(image_size: int, filters: int = 64, num_res_blocks: int = 9) -> tf.keras.Model:
    init = tf.random_normal_initializer(0.0, 0.02)
    inputs = tf.keras.layers.Input(shape=[image_size, image_size, 3])

    x = ReflectionPadding2D((3, 3))(inputs)
    x = tf.keras.layers.Conv2D(filters, 7, padding="valid", kernel_initializer=init, use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = downsample(filters * 2, 3)(x)
    x = downsample(filters * 4, 3)(x)

    for _ in range(num_res_blocks):
        x = residual_block(x, filters * 4)

    x = upsample(filters * 2, 3)(x)
    x = upsample(filters, 3)(x)

    x = ReflectionPadding2D((3, 3))(x)
    outputs = tf.keras.layers.Conv2D(3, 7, padding="valid", kernel_initializer=init, activation="tanh")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet_generator")


def build_patch_discriminator(image_size: int, filters: int = 64) -> tf.keras.Model:
    init = tf.random_normal_initializer(0.0, 0.02)
    inp = tf.keras.layers.Input(shape=[image_size, image_size, 3])
    x = downsample(filters, 4, apply_norm=False)(inp)
    x = downsample(filters * 2, 4)(x)
    x = downsample(filters * 4, 4)(x)

    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(filters * 8, 4, strides=1, kernel_initializer=init, use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.ZeroPadding2D()(x)
    out = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=init)(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="patch_discriminator")


class CycleGAN(tf.keras.Model):
    def __init__(
        self,
        generator_g: tf.keras.Model,
        generator_f: tf.keras.Model,
        discriminator_x: tf.keras.Model,
        discriminator_y: tf.keras.Model,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
    ):
        super().__init__()
        self.gen_g = generator_g
        self.gen_f = generator_f
        self.disc_x = discriminator_x
        self.disc_y = discriminator_y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def compile(
        self,
        gen_g_optimizer: tf.keras.optimizers.Optimizer,
        gen_f_optimizer: tf.keras.optimizers.Optimizer,
        disc_x_optimizer: tf.keras.optimizers.Optimizer,
        disc_y_optimizer: tf.keras.optimizers.Optimizer,
    ):
        super().compile()
        self.gen_g_optimizer = gen_g_optimizer
        self.gen_f_optimizer = gen_f_optimizer
        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer

    def generator_loss(self, fake: tf.Tensor) -> tf.Tensor:
        return self.loss_fn(tf.ones_like(fake), fake)

    def discriminator_loss(self, real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
        real_loss = self.loss_fn(tf.ones_like(real), real)
        fake_loss = self.loss_fn(tf.zeros_like(fake), fake)
        return (real_loss + fake_loss) * 0.5

    def cycle_loss(self, real: tf.Tensor, cycled: tf.Tensor) -> tf.Tensor:
        return self.lambda_cycle * tf.reduce_mean(tf.abs(real - cycled))

    def identity_loss(self, real: tf.Tensor, same: tf.Tensor) -> tf.Tensor:
        return self.lambda_identity * tf.reduce_mean(tf.abs(real - same))

    @tf.function
    def train_step(self, batch_data: Tuple[tf.Tensor, tf.Tensor]):
        real_x, real_y = batch_data

        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.gen_g(real_x, training=True)
            cycled_x = self.gen_f(fake_y, training=True)

            fake_x = self.gen_f(real_y, training=True)
            cycled_y = self.gen_g(fake_x, training=True)

            same_x = self.gen_f(real_x, training=True)
            same_y = self.gen_g(real_y, training=True)

            disc_real_x = self.disc_x(real_x, training=True)
            disc_fake_x = self.disc_x(fake_x, training=True)
            disc_real_y = self.disc_y(real_y, training=True)
            disc_fake_y = self.disc_y(fake_y, training=True)

            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.cycle_loss(real_x, cycled_x) + self.cycle_loss(real_y, cycled_y)
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        grads_gen_g = tape.gradient(total_gen_g_loss, self.gen_g.trainable_variables)
        grads_gen_f = tape.gradient(total_gen_f_loss, self.gen_f.trainable_variables)
        grads_disc_x = tape.gradient(disc_x_loss, self.disc_x.trainable_variables)
        grads_disc_y = tape.gradient(disc_y_loss, self.disc_y.trainable_variables)

        self.gen_g_optimizer.apply_gradients(zip(grads_gen_g, self.gen_g.trainable_variables))
        self.gen_f_optimizer.apply_gradients(zip(grads_gen_f, self.gen_f.trainable_variables))
        self.disc_x_optimizer.apply_gradients(zip(grads_disc_x, self.disc_x.trainable_variables))
        self.disc_y_optimizer.apply_gradients(zip(grads_disc_y, self.disc_y.trainable_variables))

        return {
            "gen_g_loss": total_gen_g_loss,
            "gen_f_loss": total_gen_f_loss,
            "disc_x_loss": disc_x_loss,
            "disc_y_loss": disc_y_loss,
        }


def save_sample_image(image_tensor: tf.Tensor, output_path: pathlib.Path):
    image = tf.cast(denormalize(image_tensor[0]), tf.uint8).numpy()
    encoded = tf.image.encode_jpeg(image)
    tf.io.write_file(str(output_path), encoded)


def train(
    photos_dir: pathlib.Path,
    vangogh_dir: pathlib.Path,
    output_dir: pathlib.Path,
    image_size: int = 256,
    batch_size: int = 1,
    epochs: int = 20,
    learning_rate: float = 2e-4,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"
    checkpoints_dir.mkdir(exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    ds_x = make_dataset(photos_dir, image_size=image_size, batch_size=batch_size, training=True)
    ds_y = make_dataset(vangogh_dir, image_size=image_size, batch_size=batch_size, training=True)
    train_ds = tf.data.Dataset.zip((ds_x, ds_y))

    gen_g = build_resnet_generator(image_size=image_size)
    gen_f = build_resnet_generator(image_size=image_size)
    disc_x = build_patch_discriminator(image_size=image_size)
    disc_y = build_patch_discriminator(image_size=image_size)

    cyclegan = CycleGAN(gen_g, gen_f, disc_x, disc_y)
    cyclegan.compile(
        gen_g_optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=0.5),
        gen_f_optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=0.5),
        disc_x_optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=0.5),
        disc_y_optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=0.5),
    )

    ckpt = tf.train.Checkpoint(
        gen_g=gen_g,
        gen_f=gen_f,
        disc_x=disc_x,
        disc_y=disc_y,
        gen_g_optimizer=cyclegan.gen_g_optimizer,
        gen_f_optimizer=cyclegan.gen_f_optimizer,
        disc_x_optimizer=cyclegan.disc_x_optimizer,
        disc_y_optimizer=cyclegan.disc_y_optimizer,
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, str(checkpoints_dir), max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f"Restored checkpoint: {ckpt_manager.latest_checkpoint}")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(train_ds):
            metrics = cyclegan.train_step(batch)
            if step % 100 == 0:
                print(
                    f"step={step} "
                    f"gen_g={float(metrics['gen_g_loss']):.4f} "
                    f"gen_f={float(metrics['gen_f_loss']):.4f} "
                    f"disc_x={float(metrics['disc_x_loss']):.4f} "
                    f"disc_y={float(metrics['disc_y_loss']):.4f}"
                )

        sample_input = next(iter(ds_x))
        sample_output = gen_g(sample_input, training=False)
        save_sample_image(sample_output, samples_dir / f"epoch_{epoch + 1:03d}.jpg")
        ckpt_path = ckpt_manager.save()
        print(f"Saved checkpoint: {ckpt_path}")

    gen_g.save(str(output_dir / "generator_photo_to_vangogh.keras"))
    print("Saved final generator model.")


def stylize_folder(
    generator_path: pathlib.Path,
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    image_size: int = 256,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = tf.keras.models.load_model(
        str(generator_path), custom_objects={"ReflectionPadding2D": ReflectionPadding2D}
    )

    for image_path in sorted(input_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = tf.io.read_file(str(image_path))
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [image_size, image_size])
        image = normalize(image)
        image = tf.expand_dims(image, axis=0)

        stylized = generator(image, training=False)
        out_img = tf.cast(denormalize(stylized[0]), tf.uint8)
        out_path = output_dir / f"{image_path.stem}_vangogh.jpg"
        tf.io.write_file(str(out_path), tf.image.encode_jpeg(out_img))
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="CycleGAN landscape generation in Van Gogh style")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train CycleGAN")
    train_parser.add_argument("--photos-dir", type=pathlib.Path, required=True)
    train_parser.add_argument("--vangogh-dir", type=pathlib.Path, required=True)
    train_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    train_parser.add_argument("--image-size", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)

    stylize_parser = subparsers.add_parser("stylize", help="Stylize a folder of landscapes")
    stylize_parser.add_argument("--generator", type=pathlib.Path, required=True)
    stylize_parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    stylize_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    stylize_parser.add_argument("--image-size", type=int, default=256)

    args = parser.parse_args()

    if args.command == "train":
        train(
            photos_dir=args.photos_dir,
            vangogh_dir=args.vangogh_dir,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
    elif args.command == "stylize":
        stylize_folder(
            generator_path=args.generator,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            image_size=args.image_size,
        )


if __name__ == "__main__":
    main()
