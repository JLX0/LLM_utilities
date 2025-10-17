import os
import subprocess
import sys
import threading
from typing import List


def execute_bash(command: str, print_progress: bool = False) -> (int, str):
    """
    Executes a bash command and returns the return code and combined output.

    Args:
        command (str): The bash command to execute.
        print_progress (bool): If True, prints output in real-time.

    Returns:
        returncode (int): The return code of the process.
        output (str): Combined stdout and stderr as a single string.
    """
    # Force unbuffered output for Python commands
    if command.strip().startswith("python"):
        command = command.replace("python", "python -u", 1)

    process = subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Ensures output is returned as strings, not bytes
        bufsize=1,  # Line-buffered output
        env=os.environ.copy(),  # Inherit environment variables
    )

    output_lines: List[str] = []  # Stores all output lines

    # Function to handle real-time printing and capturing output
    def handle_output(stream, is_stderr: bool = False):
        for line in iter(stream.readline, ""):  # Read line by line
            output_lines.append(line)  # Capture the line
            if print_progress:
                # Print to the appropriate stream (stdout or stderr)
                print(line, end="", file=sys.stderr if is_stderr else sys.stdout)
        stream.close()

    # Create threads to handle stdout and stderr concurrently
    stdout_thread = threading.Thread(target=handle_output, args=(process.stdout, False))
    stderr_thread = threading.Thread(target=handle_output, args=(process.stderr, True))

    # Start threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for the process to complete
    process.wait()

    # Wait for threads to finish
    stdout_thread.join()
    stderr_thread.join()

    # Combine all output lines into a single string
    output = "".join(output_lines)

    return process.returncode, output
