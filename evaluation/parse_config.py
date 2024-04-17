import json
import sys


def json_to_command_line(json_data):
    command_line = []
    if json_data["precision"] == "8bit":
        command_line.append(f"--bits 8")
    elif json_data["precision"] == "4bit":
        command_line.append(f"--bits 4")

    command_line.append(f"--woq_algo {json_data['quant_type']}")
    command_line.append(f"--weight_dtype {json_data['weight_dtype']}")
    command_line.append(f"--model {json_data['model']}")
    return command_line


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    with open(json_file, "r") as file:
        json_data = json.load(file)

    command_line = json_to_command_line(json_data)
    print(" ".join(command_line))


if __name__ == "__main__":
    main()
