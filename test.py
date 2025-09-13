max_end = 184.0

def fix_end(max_end):
    print(max_end)
    return int(((max_end // 4)) * 4)


print(max_end)

fixed_end = fix_end(max_end // 16)

print(fixed_end)
fixed_end = fixed_end * 16
print(f"fixed_end: {fixed_end}")

184 //16*16 