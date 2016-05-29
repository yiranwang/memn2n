define(['template/result-template', 'backbone'], function(resultTemplate, Backbone) {

    ResultView = Backbone.View.extend({
        initialize: function() {
            this.setElement("#result_container");
            this.render();
        },

        render: function() {
            this.renderContent();

            return this;
        },

        renderContent: function () {
            var confidence = 51,
                progressClass = "";

            if (confidence > 80)
                progressClass = "progress-bar-success";
            else if (confidence > 50)
                progressClass = "progress-bar-warning";
            else
                progressClass = "progress-bar-danger";

            var template = _.template(resultTemplate, {
                "answer": "answer",
                "correctAnswer": "correctAnswer",
                "confidence": confidence,
                "progressClass": progressClass
            });
            this.$el.html(template);
        }
    });

    return ResultView;
});
