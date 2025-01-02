import json

class Storage_base:
    """This class is used to read and write information in a json file"""

    def __init__(self, path):
        self.path = path
        self.information = {}

    def load_info(self):
        """
        This method loads the information from a json file.

        Args:

        Returns:

        """

        try:
            with open(self.path, "r") as file:
                self.information = json.load(file)
            print(f"Information is loaded from {self.path}.")
        except:
            print(f"No existing stored information is found in {self.path}.")

    def save_info(self):
        """
        This method saves the information to a JSON file in a nicely formatted way.

        Args:

        Returns:

        """
        with open(self.path, "w") as file:
            json.dump(self.information , file , indent=4)  # Use indent=4 for pretty formatting
        print(f"Information is saved to {self.path}.")