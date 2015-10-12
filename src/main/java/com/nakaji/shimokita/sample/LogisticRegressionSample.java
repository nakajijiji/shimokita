package com.nakaji.shimokita.sample;

import com.nakaji.shimokita.core.LogisticRegression;
import com.nakaji.shimokita.data.SampleFeatureVectors;

public class LogisticRegressionSample {
	public static void main(String[] args) {
		LogisticRegression regression = new LogisticRegression(1.0);
		regression.train(SampleFeatureVectors.generate());
	}
}
