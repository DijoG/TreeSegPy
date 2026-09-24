# Plot the threshold sweep figure from results/threshold_sweep.csv
library(ggplot2)
library(dplyr)
library(tidyr)

sweep <- read.csv("D:/TopTreeSegR/results/threshold_sweep.csv")

long <- sweep %>%
  pivot_longer(cols = c(precision, recall, f1),
               names_to = "metric",
               values_to = "value")

agg <- long %>%
  group_by(threshold, metric) %>%
  summarise(
    mean = mean(value),
    sd   = sd(value),
    .groups = "drop"
  )

optimal_t <- sweep %>%
  group_by(plot) %>%
  slice_max(f1, n = 1) %>%
  ungroup() %>%
  summarise(m = mean(threshold)) %>%
  pull(m)

p <- ggplot(agg, aes(x = threshold, y = mean, color = metric, fill = metric)) +
  geom_ribbon(aes(ymin = pmax(mean - sd, 0), ymax = pmin(mean + sd, 1)),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.2) +
  geom_vline(xintercept = optimal_t, linetype = "dashed",
             color = "grey40", linewidth = 0.4) +
  annotate("text", x = optimal_t + 0.005, y = 0.55,
           label = sprintf("optimal = %.2f", optimal_t),
           hjust = 0, size = 3.5, color = "grey30") +
  scale_color_manual(values = c("precision" = "#4477AA",
                                "recall"    = "#CC6677",
                                "f1"        = "#117733")) +
  scale_fill_manual(values = c("precision" = "#4477AA",
                               "recall"    = "#CC6677",
                               "f1"        = "#117733")) +
  scale_x_continuous(breaks = seq(0.1, 0.55, 0.05)) +
  scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.2)) +
  labs(x = "Probability threshold",
       y = "Metric",
       color = NULL, fill = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        panel.grid.minor = element_blank())

ggsave("D:/TopTreeSegR/TreeSeg_results/fig_threshold_sweep.pdf", p, width = 7, height = 4.5)
ggsave("D:/TopTreeSegR/TreeSeg_results/fig_threshold_sweep.png", p, width = 7, height = 4.5, dpi = 300)
cat("wrote TreeSeg_results/fig_threshold_sweep.{pdf,png}\n")