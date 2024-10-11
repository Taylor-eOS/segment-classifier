def divisors_under(n):
    divisors = []
    for i in range(1, 50):
        if n % i == 0:
            divisors.append(i)
    return divisors

n = input('Sample rate: ')
n = int(n)
result = divisors_under(n)
print(result)

