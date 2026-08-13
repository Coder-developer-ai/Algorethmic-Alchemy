import QuantumTree.*;

operation Main() : Result {

    let depth = 3;

    let features = [
        0.5,
        1.0,
        1.5
    ];

    let angles = [
        0.4,
        0.6,
        0.8
    ];

    return QuantumTree(
        depth,
        features,
        angles
    );
}