OPTIONAL: put your own labelled comments here to teach the models the language of real consultations.

Each CSV needs two columns:
    text,label
    "The compliance cost will crush small firms",negative
    "Please clarify the effective date",neutral
    "A welcome step for citizen privacy",positive

label must be negative, neutral or positive. Even 200-300 real comments help a lot.
They are picked up automatically the next time you run the train_* scripts.
