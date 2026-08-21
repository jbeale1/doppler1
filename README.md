This project logs data from a doppler radar traffic speed sensor (HLK-LD2415H) and separately presents views of the statistics across multiple days.

A speed histogram based on 161 days (91k traffic events) on a residential street shows a good fit to a normal distribution. Leaving off the slower side (neighbors, deliveries) there is just under 2% of traffic that falls outside the best-fit model gaussian. This forms an excess on the high-speed side that does not match a single-population normal curve model.

In the histogram I label the events under the gaussian curve (fit to the [18,36] mph interval) as Na and the points above and to the right of the curve (higher than the curve peak at 25.9 mph) as Nb. In this dataset Na (fits Gaussian) = 89683 and Nb (excess above fit on high side) = 1628. The sum (89683+1628)=91311 and there are 1628/91311 = 1.78% of that set that is the population faster than the normal fit would predict.
