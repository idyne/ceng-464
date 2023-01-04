import os

class FileIO:
    @staticmethod
    def read_recursively(path):
        result = []
        if path[-1] != "/":
            path += "/"
        dirs = os.listdir(path)
        for i in range(len(dirs)):
            dir = dirs[i]
            new_path = path + dir
            if os.path.isdir(new_path):
                result.extend(FileIO.read_recursively(new_path))
            elif dir.endswith(".txt"):
                result.append(FileIO.parse_file(new_path))

        return result

    @staticmethod
    def parse_file(path):
        return open(path).read()
