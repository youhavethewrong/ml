(ns ml.fish-test
  (:require  [clojure.test :refer :all]
             [ml.fish :refer :all]))

(testing "Produce some dark salmon evidence"
  (is
   (< (evidence-of-salmon :dark) 1)))

(testing "Produce some light salmon evidence"
  (is
   (< (evidence-of-salmon :light) 1)))
