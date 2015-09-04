(ns ml.fish)

(defn make-sea-bass
  "Sea bass are mostly long and light in color."
  []
  #{:sea-bass
    (if (< (rand) 0.2) :fat :thin)
    (if (< (rand) 0.7) :long :short)
    (if (< (rand) 0.8) :light :dark)})

(defn make-salmon
  "Salmon are mostly fat and dark in color."
  []
  #{:salmon
    (if (< (rand) 0.8) :fat :thin)
    (if (< (rand) 0.5) :long :short)
    (if (< (rand) 0.3) :light :dark)})

(defn make-sample-fish
  []
  (if (< (rand) 0.3) (make-sea-bass) (make-salmon)))

(def fish-training-data
  (for [i (range 10000)] (make-sample-fish)))

(defn probability
  "Calculates the probablility of a specific category given some attributes,
   depending on the training data."
  [attribute & {:keys [category prior-positive prior-negative data]
                :or {category nil
                     data fish-training-data}}]
  (let [by-category (if category
                      (filter category data)
                      data)
        positive (count (filter attribute by-category))
        negative (- (count by-category) positive)
        total (+ positive negative)]
    (/ positive total)))

(defn evidence-of-salmon
  [& attrs]
  (let [attr-probs (map #(probability % :category :salmon) attrs)
        class-and-attr-prob (conj attr-probs
                                   (probability :salmon))]
    (float (apply * class-and-attr-prob))))

(defn evidence-of-sea-bass
  [& attrs]
  (let [attr-probs (map #(probability % :category :sea-bass) attrs)
        class-and-attr-prob (conj attr-probs
                                   (probability :sea-bass))]
    (float (apply * class-and-attr-prob))))

(defn evidence-of-category-with-attrs
  [category & attrs]
  (let [attr-probs (map #(probability % :category category) attrs)
        class-and-attr-prob (conj attr-probs
                                   (probability category))]
    (float (apply * class-and-attr-prob))))

(def probability-dark-long-fat-is-salmon
  (let [attrs [:dark :long :fat]
        sea-bass? (apply evidence-of-sea-bass attrs)
        salmon? (apply evidence-of-salmon attrs)]
    (/ salmon?
       (+ sea-bass? salmon?))))
