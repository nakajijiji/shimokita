package com.nakaji.shimokita.core;

public interface Classifier<L> {
	L classify(FeatureVector feature);
}
