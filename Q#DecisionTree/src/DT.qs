namespace QuantumTree {

    operation QuantumTree (
        depth : Int,
        features : Double[],
        angles : Double[]
    ) : Result {

        use branches = Qubit[depth];
        use target = Qubit();

        // Branch superposition + feature encoding
        for i in 0 .. depth - 1 {
            H(branches[i]);

            let featureIndex = i % Length(features);
            Ry(features[featureIndex], branches[i]);
        }

        // Class accumulation
        H(target);

        // Level-wise quantum splits
        for i in 0 .. depth - 1 {
            Controlled Ry(
                [branches[i]],
                (angles[i], target)
            );
        }

        // Neighbor branch interactions
        if depth > 1 {
            for i in 0 .. depth - 2 {
                Controlled Ry(
                    [branches[i], branches[i + 1]],
                    (angles[i] / 2.0, target)
                );
            }
        }

        let result = M(target);

        Reset(target);
        ResetAll(branches);

        return result;
    }
    
}