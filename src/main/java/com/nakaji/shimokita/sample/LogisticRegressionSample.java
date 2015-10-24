package com.nakaji.shimokita.sample;

import com.nakaji.shimokita.data.SampleFeatureVectors;
import com.nakaji.shimokita.extention.LogisticRegression;

public class LogisticRegressionSample {
	public static void main(String[] args) {
		LogisticRegression regression = new LogisticRegression().withL2(1.0);
		regression.train(SampleFeatureVectors.generate());
	}
}
