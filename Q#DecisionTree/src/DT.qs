namespace QuantumTree {

    operation QuantumTree(
        depth : Int,
        features : Double[],
        angles : Double[]
    ) : Result {

        use branches = Qubit[depth];
        use target = Qubit();

        for i in 0 .. depth - 1 {
            H(branches[i]);

            let featureIndex = i % Length(features);

            Ry(
                features[featureIndex],
                branches[i]
            );
        }

        for i in 0 .. depth - 1 {
            Controlled Ry(
                [branches[i]],
                (angles[i], target)
            );
        }

        let result = M(target);

        Reset(target);
        ResetAll(branches);

        return result;
    }
}