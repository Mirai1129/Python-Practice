def tdee_counter(exercise_score: int, bmr: float = 0.0) -> float:
    if exercise_score == 0:
        return bmr * 1.2
    elif exercise_score == 1:
        return bmr * 1.375
    elif exercise_score == 2:
        return bmr * 1.55
    elif exercise_score == 3:
        return bmr * 1.725


def main():
    exercise_score = int(input("Enter your exercise score: "))
    bmr = float(input("Enter your BMR: "))
    tdee = tdee_counter(exercise_score, bmr)
    print(f"Exercise TDEE: {tdee}")


if __name__ == '__main__':
    main()
