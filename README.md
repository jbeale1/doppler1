This project records data from a doppler radar traffic speed sensor (HLK-LD2415H) and presents views of the statistics across multiple days.

A speed histogram based on 161 days (~91k traffic events) on a residential street shows the distribution is nearly normal, if we ignore the lower-speed tail of local arrivals and departures.
![Speeds](histogram_linear.png)

Using a log Y axis, the deviation from the normal curve is more visible, with the faster traffic extending beyond the curve. There is just under 2% of traffic that falls above and to the right of the best-fit model gaussian. This forms an excess on the high-speed side that does not match a single-population normal curve model.

![Speed histogram](hist_plot_91k.png)

In the histogram I label the set of events under the gaussian curve (fit to the [18,36] mph interval) as **Na** and the points above and to the right of the curve (higher than the curve peak at 25.9 mph) as **Nb**. In this dataset **Na** (fits Gaussian) = 89683 events and **Nb** (excess above fit on high side) = 1628. The sum (89683+1628)=91311 so there are 1628/91311 = 1.78% of the combined set that is the fraction of traffic faster than the normal fit would predict.
