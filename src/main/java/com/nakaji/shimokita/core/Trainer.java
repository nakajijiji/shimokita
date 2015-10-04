package com.nakaji.shimokita.core;

import java.util.List;
import java.util.Map;

public interface Trainer {
	<L> Classifier<L> train(Map<L, List<FeatureVector>> featureMap);
}
