import csv

with open("class_counts.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

zero_train = [
    row for row in rows
    if int(row["train"]) == 0
]

print("CLASSES WITH ZERO TRAINING INSTANCES")
print("=" * 50)

for row in zero_train:
    print(
        f"{row['name']}: "
        f"train={row['train']}, "
        f"valid={row['valid']}, "
        f"test={row['test']}"
    )

print(f"\nTotal classes with zero training instances: {len(zero_train)}")