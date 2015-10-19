(ns ml.encog
(:use [enclog nnets training]))

(def dataset
  (let [xor-input [[0.0 0.0]
                   [1.0 0.0]
                   [0.0 1.0]
                   [1.0 1.0]]
        xor-ideal [[0.0]
                   [1.0]
                   [1.0]
                   [0.0]]]
    (data :basic-dataset xor-input xor-ideal)))

(defn round-output
  "Round outputs to nearest integer."
  [output]
  (mapv #(Math/round ^Double %) output))

(defn train-network
  [network data trainer-algo]
  (let [trainer (trainer trainer-algo
                         :network network
                         :training-set data)]
    (train trainer 0.01 1000 [])))

(defn run-network
  [network input]
  (let [input-data (data :basic input)
        output (.compute network input-data)
        output-vec (.getData output)]
    (round-output output-vec)))

(def mlp (network (neural-pattern :feed-forward)
                  :activation :sigmoid
                  :input 2
                  :output 1
                  :hidden [3]))

(def MLP (train-network mlp dataset :back-prop))


(def elman-network (network (neural-pattern :elman)
                            :activation :sigmoid
                            :input 2
                            :output 1
                            :hidden [3]))

(def EN (train-network elman-network dataset :resilient-prop))

