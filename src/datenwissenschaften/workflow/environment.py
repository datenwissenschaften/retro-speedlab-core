import flyte

IMAGE = "retro-speedlab"

speedlab = flyte.TaskEnvironment(
    name="retro-speedlab",
    image=flyte.Image.from_ref_name(IMAGE),
    resources=flyte.Resources(gpu=1, shm="2Gi"),
)
