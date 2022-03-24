if __name__ == '__main__':
    with open('./run.sh', 'r') as f:
        text = f.read()
        text.replace('\r', '')

    with open('./run.sh', 'w') as f:
        f.write(text)
