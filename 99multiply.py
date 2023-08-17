# for i in range(1, 10):
#     for j in range(1,i+1):
#         print(f"{i}x{j}={i * j}", end=" ")
#     print()
# for i in range(11):
#     if i <= 5:
#         print(i * "* ")
#     else:
#         print((10-i) * "* ")
count = 0
black_girl_age = 25
while count < 4:
    guess = input("猜猜黑姑娘年龄>:")
    if guess.isdigit():
        guess = int(guess)
    else:
        print("不识别的年龄，请重新输入......")
        continue
    if guess > black_girl_age:
        print("猜大了，往小了试...")
    elif guess < black_girl_age:
        print("猜小了，往大了试...")
    else:
        print("恭喜你猜对了，黑姑娘送你一晚....")
        break
    count += 1