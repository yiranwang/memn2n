define(["backbone"], function(Backbone) {
    var Main = Backbone.Model.extend({

        defaults: {
            "hop": 1,
            "story": [],
            "question": "",
            "answer": "",
            "answerProbability": 0,
            "memoryProbabilities": []
        }

    });

    return Main;
});
