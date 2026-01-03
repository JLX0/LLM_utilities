from collections.abc import Callable
import json
import os
from typing import Any
from typing import TypeVar


F = TypeVar("F", bound=Callable[..., Any])


class Storage_base:
    """This class is used to read and write information in a json file"""

    def __init__(self, path: str, debug: bool = False) -> None:
        self.path = path
        self.information: dict[str, Any] = {}
        self.debug = debug

    def load_info(self) -> None:
        """
        This method loads the information from a json file.

        Args:

        Returns:

        """

        try:
            with open(self.path) as file:
                self.information = json.load(file)
            if self.debug:
                print(f"Information is loaded from {self.path}.")
        except Exception:
            if self.debug:
                print(f"No existing stored information is found in {self.path}.")

    def save_info(self) -> None:
        """
        This method saves the information to a JSON file in a nicely formatted way.

        Args:

        Returns:

        """
        with open(self.path, "w") as file:
            json.dump(self.information, file, indent=4)  # Use indent=4 for pretty formatting
        if self.debug:
            print(f"Information is saved to {self.path}.")

    @classmethod
    def auto_load_save(cls, method: F) -> F:
        """
        Decorator to automatically call self.load_info() before the method
        and self.save_info() after the method.
        """

        def wrapper(self: "Storage_base", *args: Any, **kwargs: Any) -> Any:
            self.load_info()
            result = method(self, *args, **kwargs)
            self.save_info()
            return result

        return wrapper  # type: ignore[return-value]


def save_python_code(python_code: str, file_path: str) -> None:
    # Extract the directory path from the file_path
    directory = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save the Python code to the file
    with open(file_path, "w") as file:
        file.write(python_code)

    print(f"Python code has been saved to {file_path}")
