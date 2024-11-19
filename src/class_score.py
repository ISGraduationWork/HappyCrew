def class_score(neu, hap, sad, sur, ang):
    n = (neu * 0.2) + hap - sad + sur - ang
    if n > 115:
        score = 100
    elif n < -95:
        score = 0
    else:
        x = n + 95
        # x: 0~210
        score = x * 100 / 210
    return score