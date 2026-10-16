# Part 3: one prompt per required theme
PART3_PROMPTS = {
    "mystery": "One night, Mia heard a strange knock on the window. When she looked outside, nobody was there, but a small box was sitting on the grass.",
    "discovery": "Tom was digging in the garden when his spade hit something hard. He pulled it out and saw that it was glowing.",
    "unusual_character": "Pip was a tiny dragon who was afraid of fire. Every day he hid behind the big rock so nobody would find out.",
    "problem": "The little boat had a hole in it, and the water was coming in fast. Sam and his sister had to fix it before the sun went down.",
    "unexpected_ending": "Lily worked hard all week to bake the biggest cake in the village. When she carried it to the table, she tripped, and then something very surprising happened.",
}

# Part 4: plot-twist openings
PART4_PROMPTS = {
    "exam": "A girl named Anna opened her exam paper and froze. The answers to tomorrow's questions were already written on it.",
    "robot": "The old robot sat alone in the shop. One day it received a message that said, I am you, from the future.",
    "detective": "The detective looked at the body on the floor. The murder had not happened yet, but he knew exactly who would do it.",
    "village": "Everyone in the village woke up and felt very strange. Nobody could remember what had happened the day before.",
    "letter": "Ben found a letter on his doorstep with his name on it. The date on the letter was one hundred years from now.",
}

CONFIGS = {
    "greedy": dict(strategy="greedy"),
    "temp0.5": dict(strategy="sample", temperature=0.5),
    "temp0.8": dict(strategy="sample", temperature=0.8),
    "temp1.2": dict(strategy="sample", temperature=1.2),
    "topk10": dict(strategy="sample", temperature=1.0, top_k=10),
    "topk40": dict(strategy="sample", temperature=1.0, top_k=40),
}
PART4_CONFIGS = ["temp0.8", "topk40"]
