define(['template/analysis-template', 'backbone', 'view/analysis-view-word'], function(analysisTemplate, Backbone, AnalysisWordView) {

    var AnalysisView = Backbone.View.extend({
        initialize: function () {
            this.setElement("#analysis_container");
            this.render();

            this.listenTo(this.model, "change:answer", this._onAnswer);
        },

        render: function () {
            var template = _.template(analysisTemplate[0]);
            this.$el.html(template);

            this.renderContent();

            this.$wordView = new AnalysisWordView({ model: this.model }).render().$el;
            this.$root.append(this.$wordView);

            return this;
        },

        renderContent: function () {
            this.$root = this.$el.find(".col-md-6");
            this.$table = this.$el.find(".hop-table");

            this.renderSentences();
        },

        renderSentences: function () {
            var sentences = this.model.get('story'),
                rowTemplate = analysisTemplate[1],
                headerTemplate = _.template(analysisTemplate[2]),
                self = this;

            this.$table.append(headerTemplate);
            var probs = this.model.get('memoryProbabilities');

            _.each(sentences, function(val, idx) {
                var prob = probs[idx],
                    row = _.template(rowTemplate, {
                        "sentence": val,
                        "perc1": prob[0].toFixed(3),
                        "perc2": prob[1].toFixed(3),
                        "perc3": prob[2].toFixed(3)
                    });
                self.$table.append(row);
            });
        },

        _onAnswer: function () {
            this.$table.empty();
            this.renderContent();
        }
    });

    return AnalysisView;
});
