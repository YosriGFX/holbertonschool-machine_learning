#!/usr/bin/env python3
'''script that takes in input from the user
with the prompt Q: and prints A: as a response'''


if __name__ == "__main__":
    while True:
        print(
            'Q:', end=' '
        )
        Question = input()
        if Question.lower() in [
            'bye',
            'goodbye',
            'quit',
            'exit'
        ]:
            Answer = 'Goodbye'
            print('A: {}'.format(Answer))
            break
        Answer = ''
        print(
            'A: {}'.format(
                Answer
            )
        )
