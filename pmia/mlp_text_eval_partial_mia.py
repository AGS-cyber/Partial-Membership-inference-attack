"""20 Newsgroups/MLP compatibility entry point."""

if __package__:
    from .run_experiment import preset_entrypoint
else:
    from run_experiment import preset_entrypoint


if __name__ == "__main__":
    raise SystemExit(preset_entrypoint("newsgroups", "mlp"))
