# interact

This repository is a collection of the code I wrote while researching with the [Mechanical Systems Control Lab](https://msc.berkeley.edu/). The goal is to generate routable graphs from collections of GPS traces from the [INTERACTION dataset](http://interaction-dataset.com/).

## Algorithms/Results

The processing of data and graph generation algorithm was taken from the Microsoft paper by [Cao and Krumm](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/maps-from-GPS.pdf).

The raw data, when plotted, looks like:

![](images/before.png)


By simulating an attractive gravitational force between traces and using a graph generation algorithm, the following map is created:

![](images/after.png)

## Improvements
There are still many improvements to be made:

- Cleaning up graph where multiple roads merge or split
- Improving upon preprocessing of data
- Conduct lots of testing to determine optimal parameters
