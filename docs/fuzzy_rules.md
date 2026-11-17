# Fuzzy Rule Base

There are 128 rules because 7 inputs each have 2 terms:

2^7 = 128.

Terms:
- LOW
- HIGH

Representative rules for explanation:

R1:
IF HR is HIGH AND Temp is HIGH AND SBP is LOW AND Resp is HIGH AND O2Sat is LOW AND Lactate is HIGH AND WBC is HIGH THEN risk is high.

R2:
IF HR is LOW AND Temp is LOW AND SBP is HIGH AND Resp is LOW AND O2Sat is HIGH AND Lactate is LOW AND WBC is LOW THEN risk is low.

The actual neuro-fuzzy model learns continuous Sugeno consequents instead of assigning a manually fixed medical diagnosis to each rule. These are computational rules, not clinical recommendations.
