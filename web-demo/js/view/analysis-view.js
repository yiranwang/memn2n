define(['template/analysis-template', 'backbone'], function(analysisTemplate, Backbone) {

    AnalysisView = Backbone.View.extend({
        initialize: function () {
            this.setElement("#analysis_container");
            this.render();
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
            var sentences = ['a', 'b', 'c', 'd'],
                rowTemplate = analysisTemplate[1],
                self = this;

            var probs =   [[  6.34466469e-01,   9.52042341e-01,   9.86093879e-01],
                           [  6.83657378e-02,   8.64849426e-03,   2.61654658e-03],
                           [  1.47958507e-03,   3.54395997e-05,   1.72178170e-06],
                           [  3.07917828e-03,   1.00669975e-04,   5.96504242e-06]];

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
        }
    });

    return AnalysisView;
});
