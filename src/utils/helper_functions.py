"""Utility helpers for the project: readable console output and step timing."""

import time

LINE_WIDTH = 60


def print_header(title: str) -> None:
    """Print a title framed by lines, used at the start and end of a run."""
    print("=" * LINE_WIDTH)
    print(f"  {title}")
    print("=" * LINE_WIDTH)


def print_info(label: str, value) -> None:
    """Print an indented 'label  value' line, padding the label so the values line up."""
    print(f"    {label:<24} {value}")


def start_step(name: str) -> float:
    """Print the name of a pipeline step and return the time it started."""
    print(f"\n>>> {name}")
    return time.perf_counter()


def end_step(start_time: float) -> None:
    """Print how many seconds have passed since start_time."""
    elapsed = time.perf_counter() - start_time
    print(f"    done in {elapsed:.2f}s")


if __name__ == "__main__":
    print_header("Helper function demo")
    t = start_step("Sleeping for half a second")
    time.sleep(0.5)
    end_step(t)
    print_info("Example value", 42)
