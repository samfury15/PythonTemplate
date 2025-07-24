import argparse

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x, y): return x / y if y != 0 else "Error: Division by zero"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple CLI Calculator")
    parser.add_argument("x", type=float, help="First number")
    parser.add_argument("op", type=str, choices=["+", "-", "*", "/"], help="Operation")
    parser.add_argument("y", type=float, help="Second number")
    args = parser.parse_args()

    operations = {"+": add, "-": sub, "*": mul, "/": div}
    result = operations[args.op](args.x, args.y)
    print(f"{args.x} {args.op} {args.y} = {result}")
