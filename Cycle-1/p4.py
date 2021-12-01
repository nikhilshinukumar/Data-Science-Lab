def gcd(p,q):
    while q != 0:
        p, q = q, p%q
    return p
def is_coprime(x, y):
    if(gcd(x, y) == 1):
        print("Coprime")


num1 = int(input("Enter First Number:"))
num2 = int(input("Enter Second Number:"))

is_coprime(num1,num2)