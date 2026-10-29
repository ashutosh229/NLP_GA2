# Q3 Results — Abstractive Text Summarization

## Method and configuration

A from-scratch PyTorch Transformer encoder-decoder was trained on CNN/DailyMail 3.0.0. The pretrained `bert-base-uncased` asset was used only as a tokenizer.

- Tokenizer: `bert-base-uncased`; article/summary truncation: 512/128.
- Model: 2 encoder and 2 decoder layers, d_model=128, heads=4, FFN=512, dropout=0.1.
- Parameters: 8,769,850; data counts train/validation/test: 287,113/13,368/11,490.
- Best epoch: 8; best validation loss: 5.1243; training time: 5.40 h.
- Adam betas (0.9, 0.98), epsilon 1e-9; warmup 4000, schedule scale 0.5; effective batch 16.
- Greedy autoregressive generation with a 12-token minimum and no-repeat trigram constraint. Training uses teacher forcing and padding-ignored loss.

## ROUGE F1 (percentage points)

| System | Test examples | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---:|---:|---:|---:|---:|
| Lead-3 (assignment reference) | 11,490 | 40.06 | 17.48 | 25.02 | 36.36 |
| Lead-3 (this run) | 11,490 | 39.91 | 17.38 | 24.90 | 36.20 |
| Scratch Transformer | 11,490 | 22.84 | 4.37 | 15.94 | 21.38 |

Full official test split evaluated.

## Artifacts

The ZIP includes the best model weights, a resumable optimizer/scaler checkpoint, tokenizer files, per-epoch losses, generated/reference predictions for each test example, ROUGE results, the loss curve, and a cross-attention heatmap. The attention plot averages heads from the last decoder layer.

## Qualitative examples

### Example 1

**Article excerpt:** (CNN) I see signs of a revolution everywhere. I see it in the op-ed pages of the newspapers, and on the state ballots in nearly half the country. I see it in politicians who once preferred to play it safe with this explosive issue but are now willing to stake their political futures on it. I see the revolution in the eyes of sterling scientists, previously reluctant to dip a toe into this heavily stigmatized world, who are diving in head first. I see it in the new surgeon general who cites data showing just how helpful it can be. I see a revolution in the attitudes of everyday Americans. For the first time a majority, 53%, favor its legalization, with 77% supporting it for medical purposes. Support for legalization has risen 11 points in the past few years alone. In 1969, the first time Pew asked the question about legalization, only 12% of the nation was in favor. I see a revolution tha...

**Reference:** CNN's Dr. Sanjay Gupta says we should legalize medical marijuana now . / He says he knows how easy it is do nothing "because I did nothing for too long"

**Lead-3:** (CNN) I see signs of a revolution everywhere. / I see it in the op-ed pages of the newspapers, and on the state ballots in nearly half the country. / I see it in politicians who once preferred to play it safe with this explosive issue but are now willing to stake their political futures on it.

**Generated:** the first time of the u. s. she says she says the uy says she ' s. he says she is a " the first - term " she says. she ' d been a " she ' t - - - term. she has a new law.

### Example 2

**Article excerpt:** He looks barely teenage. But this child has amassed thousands of Twitter followers with his pictorial updates of 'gang life'. The baby-faced boy from Memphis, Tennessee, poses with guns, cash, and bags of what looks like marijuana. Scroll down for video . Baby-faced: This little boy has amassed more than 3,000 followers on Twitter with pictures like these . In many pictures he is smoking suspicious substances, with captions such as 'High Life' Backlash: The boy, from Memphis, has prompted a wave of critics calling his stunts 'sad' In one video he laughs and points the gun at the camera in an apparent attempt to look menacing - as adults laugh in the background. In others, he is pictured blowing smoke, with the caption: 'High Life'. Tweets include the phrases, 'I need a bad b****', 'f*** da police', and 'gang sh** n****'. Some feature sexual references, such as: 'quit suckin my d***'. As ...

**Reference:** Child has amassed thousands of Twitter followers with 'gang life' photos . / In one video he points gun at camera as adults look on unfazed . / His tweets have prompted backlash with calls for intervention .

**Lead-3:** He looks barely teenage. / But this child has amassed thousands of Twitter followers with his pictorial updates of 'gang life'. / The baby-faced boy from Memphis, Tennessee, poses with guns, cash, and bags of what looks like marijuana.

**Generated:** the boy has been seen on twitter and his own ' s facebook page. he has been posted on twitter to the video of the video.

### Example 3

**Article excerpt:** New Jersey Governor Chris Christie wasn't looking too presidential Tuesday night when he got into a heated debate with a veteran teacher at a town hall meeting. And now the state's largest teacher's union is calling him out for his 'bullying' behavior. 'He's always taken a very nasty and disrespectful tone with teachers and other individuals who dare to question him at these events,' Steve Wollmer of the NJ Education Association told NJ.com. 'It's the one thing that never seems to change.' Scroll Down for Video . Not being nice: New Jersey Gov Chris Christie (left) is being called a bully for the way he interacted with a teacher (Kathy Mooney, right) at a Tuesday night town hall meeting . That sentiment doesn't ring well with Christie's ambitions to run in the Republican presidential primaries next year. Tuesday night, Christie appeared at Kenilworth Town Hall to take questions from a gr...

**Reference:** The presidential hopeful held a town hall meeting in Kenilworth on Tuesday . / During the meeting, high school English teacher Kathy Mooney got up to ask the governor a question about pensions . / She asked why he didn't seek a higher legal settlement in a case with ExxonMobil that would have contributed to the state's pension system . / Christie responded by repeatedly asking how much Mooney knew about the deal instead of answering her question .

**Lead-3:** New Jersey Governor Chris Christie wasn't looking too presidential Tuesday night when he got into a heated debate with a veteran teacher at a town hall meeting. / And now the state's largest teacher's union is calling him out for his 'bullying' behavior. / 'He's always taken a very nasty and disrespectful tone with teachers and other individuals who dare to question him at these events,' Steve Wollmer of the NJ Education Association told NJ.com.

**Generated:** christie ' s governor says he would have a ' s not be a ' the moon ' s ' s decision to be a new jersey state. christie ' i ' m not to be the moon, ' m a new york state ' m going to be able to be ' m just a year...

### Example 4

**Article excerpt:** YouTube star Cassey Ho has hit back at critics with a powerful and provocative new video, highlighting the cruel comments left by viewers of her fitness-focused clips who accuse the trim and toned online icon of being everything from ‘too fat’ to ‘ugly’ to ‘pudgy’. Cassey's YouTube channel, Blogilates, has over two million subscribers, so fans may have been surprised to learn that the negative comments left under the California resident’s fitness videos still manage to get under her skin – and have a much greater impact on her than any of the positive messages sent to her by her fans. But though mean and critical messages may have put a dent in the 28-year-old’s self-esteem, the workout pro is getting the last word with her latest video, The 'Perfect' Body, which examines just whether conforming to society's standards is the key to happiness, while highlighting some of the cruel comments...

**Reference:** Cassey Ho boasts over two million subscribers on her YouTube channel Blogilates . / The 28-year-old receives hundreds of comments a day telling her that she needs to lose weight .

**Lead-3:** YouTube star Cassey Ho has hit back at critics with a powerful and provocative new video, highlighting the cruel comments left by viewers of her fitness-focused clips who accuse the trim and toned online icon of being everything from ‘too fat’ to ‘ugly’ to ‘pudgy’. / Cassey's YouTube channel, Blogilates, has over two million subscribers, so fans may have been surprised to learn that the negative comments left under the California resident’s fitness videos still manage to get under her skin – and have a much greater impact on her than any of the positive messages sent to her by her fans. / But though mean and critical messages may have put a dent in the 28-year-old’s self-esteem, the workout pro is getting the last word with her latest video, The 'Perfect' Body, which examines just whether conforming to society's standards is the key to happiness, while highlighting some of the cruel comments she has received from viewers of her YouTube videos.

**Generated:** the video of the video shows her body is a video of her body. the video has been posted on youtube video of a video on youtube. the youtube video shows that she has been ' s body.

### Example 5

**Article excerpt:** British taekwondo fighter Aaron Cook has confirmed he plans to compete for Moldova at the 2016 Olympics in Rio. Dorset-born Cook, 24, who was overlooked for the Great Britain taekwondo squad at London 2012, applied for citizenship after receiving funding from Moldovan billionaire Igor Iuzefovici and has now received his passport from the small eastern European state. 'I will soon begin a new journey representing the Republic of Moldova at all International competitions and hopefully the Rio Olympic games and beyond, should I qualify,' Cook wrote on his Facebook page. Three time European champion Aaron Cook (right) has refused to fight for Great Britain  since May 2012 . The British taekwondo star has been granted citizenship by Moldova and plans to fight for them in Rio 2016 . 'Although I am upset and disappointed I will not represent my country of birth at another major championships, I...

**Reference:** Aaron Cook was overlooked by Team GB for the London Olympics . / Taekwondo star has received citizenship from Moldova and plans to fight for them at the Rio 2016 Games . / The British Olympic Association could yet block the move .

**Lead-3:** British taekwondo fighter Aaron Cook has confirmed he plans to compete for Moldova at the 2016 Olympics in Rio. / Dorset-born Cook, 24, who was overlooked for the Great Britain taekwondo squad at London 2012, applied for citizenship after receiving funding from Moldovan billionaire Igor Iuzefovici and has now received his passport from the small eastern European state. / 'I will soon begin a new journey representing the Republic of Moldova at all International competitions and hopefully the Rio Olympic games and beyond, should I qualify,' Cook wrote on his Facebook page.

**Generated:** former british olympic champion will be the first time for the olympic games. he has been named as a ' s first time since 2008.

## Limitations

A limited-capacity model trained from scratch may trail the extractive Lead-3 baseline. Review generated examples for factual errors, omissions, and repetition. Attention weights are not causal explanations.
