def solution(N):
    # N = list(str(N))
    # N.sort(reverse = True)
    # ans = int("".join(N))
    # if 0 <= ans<= 1000000000:
    #     return ans
    # else:
    #     return -1
    buckets = [0] * 10
    while N:
        num = N % 10
        N //= 10
        buckets[num] += 1
    ans = ""
    for i in range(9, -1, -1):
        if buckets[i]:
            ans += str(i) * buckets[i]
    return int(ans)

print(solution(213))
print(solution(553))