import os

from invoke import task


VERBOSE = False

PROJECT_DIR = os.getcwd()
MODEL_NAME = "distilbert-base-nli-stsb-mean-tokens"


def _build_pytorch(ctx):
    ctx.run("docker build -t pytorch:slim build-pytorch/")
    ctx.run(f"docker run --name build-pytorch --rm -v {PROJECT_DIR}/build-pytorch:/output pytorch:slim")


def _build_cupy(ctx):
    ctx.run("docker build -t cupy:slim build-cupy/")
    ctx.run(f"docker run --name build-cupy --rm -v {PROJECT_DIR}/build-cupy:/output cupy:slim")


def _download_model(ctx):
    ctx.run(f"mkdir -p {PROJECT_DIR}/model/")

    if os.exists(f"{PROJECT_DIR}/model/{MODEL_NAME}"):
        return
    with ctx.cd(f"{PROJECT_DIR}/model"):
        ctx.run("git lfs install")
        ctx.run(f"git clone https://huggingface.co/sentence-transformers/{MODEL_NAME}")


def _docker_build(ctx):
    ctx.run("DOCKER_BUILDKIT=1")
    ctx.run("docker build -t spacygpu:latest --squash .")


@task
def build(ctx):
    with ctx.cd(PROJECT_DIR):
        _build_pytorch(ctx)
        _build_cupy(ctx)
        _download_model(ctx)
        _docker_build(ctx)
