if __name__ == '__main__':
    with open("data/text8/text8", "r") as f:
        data = f.read()

        n = len(data)
        train = data[:int(n * 0.9)]
        valid = data[int(n * 0.9):int(n * 0.95)]
        test = data[int(n * 0.95):]

        with open("data/text8/train.txt", "w") as f:
            f.write(train)

        with open("data/text8/valid.txt", "w") as f:
            f.write(valid)

        with open("data/text8/test.txt", "w") as f:
            f.write(test)

        print("✅ Done splitting: train.txt, valid.txt, test.txt created.")