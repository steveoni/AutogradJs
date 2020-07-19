/**
 * Most of the utility function are drawn from convnet.js
 */
var utils = {};

(function(global){

    // Utility fun
    function assert(condition, message) {
        // from http://stackoverflow.com/questions/15313418/javascript-assert
        if (!condition) {
            message = message || "Assertion failed";
            if (typeof Error !== "undefined") {
                throw new Error(message);
            }
            throw message; // Fallback
        }
    }

    var return_v = false;
    var v_val = 0.0;
    var gaussRandom = function () {
        if (return_v) {
            return_v = false;
            return v_val;
        }
        var u = 2 * Math.random() - 1;
        var v = 2 * Math.random() - 1;
        var r = u * u + v * v;
        if (r == 0 || r > 1) return gaussRandom();
        var c = Math.sqrt(-2 * Math.log(r) / r);
        v_val = v * c; // cache this
        return_v = true;
        return u * c;
    }

    var randf = function (a, b) { return Math.random() * (b - a) + a; }
    var randi = function (a, b) { return Math.floor(Math.random() * (b - a) + a); }
    var randn = function (mu, std) { return mu + gaussRandom() * std; }

    // helper function returns array of zeros of length n
    // and uses typed arrays if available
    var zeros = function (n) {
        if (typeof (n) === 'undefined' || isNaN(n)) { return []; }
        if (typeof ArrayBuffer === 'undefined') {
            // lacking browser support
            var arr = new Array(n);
            for (var i = 0; i < n; i++) { arr[i] = 0; }
            return arr;
        } else {
            return new Float64Array(n);
        }
    }

    var RandMat = function (n, d, mu, std) {
        var m = new Tensor(n, d);
        fillRandn(m, mu, std);
        //fillRand(m,-std,std); // kind of :P
        return m;
    }

    // Mat utils
    // fill matrix with random gaussian numbers
    var fillRandn = function (w, mu, std) { for (var i = 0, n = w.length; i < n; i++) { w[i] = randn(mu, std); } }

    global.gaussRandom = gaussRandom
    global.zeros = zeros
    global.assert  =  assert
    global.fillRandn =  fillRandn
    global.RandMat  = RandMat
    global.randi  = randi
    global.randf = randf
    global.randn = randn
})(utils)
