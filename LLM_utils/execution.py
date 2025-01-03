import subprocess
import sys
import os

def execute_bash(command, print_progress=False):
    """
    Executes a bash command and returns the return code and combined output.

    Args:
        command (str): The bash command to execute.
        print_progress (bool): If True, prints output in real-time.

    Returns:
        returncode (int): The return code of the process.
        output (str): Combined stdout and stderr as a single string.
    """
    process = subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Ensures output is returned as strings, not bytes
        env = os.environ.copy()  # Ensure the subprocess inherits updated environment variables
    )

    output_lines = []  # Stores all output lines

    # Function to handle real-time printing and capturing output
    def handle_output(stream, is_stderr=False):
        for line in stream:
            output_lines.append(line)  # Capture the line
            if print_progress:
                # Print to the appropriate stream (stdout or stderr)
                print(line, end="", file=sys.stderr if is_stderr else sys.stdout)

    # Handle stdout and stderr in real-time
    handle_output(process.stdout)
    handle_output(process.stderr, is_stderr=True)

    # Wait for the process to complete
    process.wait()

    # Combine all output lines into a single string
    output = "".join(output_lines)

    return process.returncode, output