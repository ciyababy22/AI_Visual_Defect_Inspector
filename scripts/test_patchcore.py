from anomalib.models import Patchcore


def main():
    print("Creating PatchCore model...")

    model = Patchcore(
        backbone="wide_resnet50_2",
    )

    print("PatchCore model created successfully.")
    print(f"Model type: {type(model).__name__}")


if __name__ == "__main__":
    main()