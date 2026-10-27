from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable


@traceable(name="smoke_test_agent", run_type="chain", tags=["smoke"])
def hello(name: str) -> str:
    return f"hello {name}"


if __name__ == "__main__":
    print(hello("asterline"))
