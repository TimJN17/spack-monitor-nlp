

"""

This file is designed to help the "llnlproject.py" file.

"""

import github

# Function to get a list of the contents in the repository path
def git_repos_contents(user_name, repository):
    # instantiate a dictionary for the json files
    raw_url_list = []

    # Create a github object
    g = github.Github()

    # Access a user's github
    user = g.get_user(user_name)

    # Access  specific repository
    error_repository = user.get_repo(f"{repository}")

    # Get the content List
    if repository == "error_analysis":
        data = error_repository.get_contents("data")
        for item in data:
            # if "errors-" and ".json" in item.path:
            if "errors-" in item.path:
                # Here 'raw' is a list of dictionaries just like when accessing the url directly
                print(f"checking path: {item.path}")

                # Append the newly found json list of dicts ot the final dict
                raw_url_list.append(item.download_url)

    elif repository == "spack-monitor-nlp":
        data2 = error_repository.get_contents("docs")
        for item in data2:
            if "meta" in item.path:
                # Here 'raw' is a list of dictionaries just like when accessing the url directly
                print(f"checking path: {item.path}")

                # Append the newly found json list of dicts ot the final dict
                raw_url_list.append(item.download_url)

    return raw_url_list

# Function to get .py Files
def git_python_contents(user_name, repository):

    # Create a github object
    g = github.Github()

    # Access a user's github
    user = g.get_user(user_name)

    # Access  specific repository
    error_repository = user.get_repo(f"{repository}")

    # Get the content List
    data = error_repository.get_contents("data")
    for item in data:
        if ".py" in item.path:
            # Here 'raw' is a list of dictionaries just like when accessing the url directly
            print(f"checking path: {item.path}")

            # write the file out
            with open(item.path+'.py', mode="w+") as f:
                f.write(item.decoded_content.decode())
            f.close()

    return -1

