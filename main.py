import argparse
import sys
from typing import Union, Callable

def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y

def sub(x: float, y: float) -> float:
    """Subtract y from x."""
    return x - y

def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

def div(x: float, y: float) -> Union[float, str]:
    """Divide x by y with zero division check."""
    if y == 0:
        return "Error: Division by zero"
    return x / y

def power(x: float, y: float) -> float:
    """Raise x to the power of y."""
    return x ** y

def modulo(x: float, y: float) -> Union[float, str]:
    """Calculate x modulo y."""
    if y == 0:
        return "Error: Modulo by zero"
    return x % y

def validate_number(value: str) -> float:
    """Validate and convert string to float."""
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid number")

def format_result(result: Union[float, str]) -> str:
    """Format the result for display."""
    if isinstance(result, str):
        return result
    return f"{result:.6f}".rstrip('0').rstrip('.')

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced CLI Calculator with multiple operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py 5 + 3
  python main.py 10 / 2
  python main.py 2 ^ 3
  python main.py 17 % 5
        """
    )
    parser.add_argument("x", type=validate_number, help="First number")
    parser.add_argument("op", type=str, choices=["+", "-", "*", "/", "^", "%"], 
                       help="Operation (+, -, *, /, ^, %)")
    parser.add_argument("y", type=validate_number, help="Second number")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show detailed calculation information")
    
    args = parser.parse_args()

    operations: dict[str, Callable] = {
        "+": add, 
        "-": sub, 
        "*": mul, 
        "/": div, 
        "^": power, 
        "%": modulo
    }
    
    try:
        result = operations[args.op](args.x, args.y)
        formatted_result = format_result(result)
        
        if args.verbose:
            print(f"Operation: {args.x} {args.op} {args.y}")
            print(f"Result: {formatted_result}")
            if isinstance(result, float):
                print(f"Type: {type(result).__name__}")
        else:
            print(f"{args.x} {args.op} {args.y} = {formatted_result}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
