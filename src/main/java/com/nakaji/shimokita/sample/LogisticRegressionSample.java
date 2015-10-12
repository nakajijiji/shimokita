package com.nakaji.shimokita.sample;

import com.nakaji.shimokita.core.LogisticRegression;
import com.nakaji.shimokita.data.SampleFeatureVectors;

public class LogisticRegressionSample {
	public static void main(String[] args) {
		LogisticRegression regression = new LogisticRegression(0.1);
		regression.train(SampleFeatureVectors.generate());
	}
}
