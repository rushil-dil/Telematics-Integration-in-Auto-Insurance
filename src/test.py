from pydantic import BaseModel


class Model(BaseModel):
    f: str



def print_hi(input: Model) -> None:
    print(type(input.f))
    print(input.f)



if __name__ == '__main__':
    print_hi(5)