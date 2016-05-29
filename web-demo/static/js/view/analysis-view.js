define(['template/analysis-template', 'backbone'], function(analysisTemplate, Backbone) {

    AnalysisView = Backbone.View.extend({
        initialize: function () {
            this.setElement("#analysis_container");
            this.render();

            this.listenTo(this.model, "change:answer", this._onAnswer);
        },

        render: function () {
            this.renderContent();

            return this
        },

        renderContent: function () {
            var template = _.template(analysisTemplate[0]);
            this.$el.html(template);

            this.$table = this.$el.find(".table");

            this.renderSentences();
        },

        renderSentences: function () {
            var sentences = this.model.get('story'),
                rowTemplate = analysisTemplate[1],
                self = this;

            var probs = this.model.get('memoryProbabilities');

            _.each(sentences, function(val, idx) {
                var prob = probs[idx],
                    row = _.template(rowTemplate, {
                        "sentence": val,
                        "perc1": prob[0],
                        "perc2": prob[1],
                        "perc3": prob[2]
                    });
                self.$table.append(row);
            });
        },

        _getProbabilities: function (len) {
            var probs = [],
                hops = 3;

            for (i=0; i<len; i++) {
                var senprob = [];
                for (j=0; j<hops; j++) {
                    senprob.push(Math.random());
                }

                probs.push(senprob);
            }

            return probs;
        },

        _onAnswer: function () {
            this.$el.empty();
            this.renderContent();
        }
    });

    return AnalysisView;
});
