// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef Tempus_SecondOrderResidualModelEvaluator_impl_hpp
#define Tempus_SecondOrderResidualModelEvaluator_impl_hpp

namespace Tempus {

template <typename Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
SecondOrderResidualModelEvaluator<Scalar>::createInArgs() const {
#ifdef VERBOSE_DEBUG_OUTPUT
  *out_ << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  typedef Thyra::ModelEvaluatorBase MEB;

  MEB::InArgsSetup<Scalar> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.set_Np(transientModel_->Np());
  inArgs.setSupports(MEB::IN_ARG_x);

  return inArgs;
}

template <typename Scalar>
Thyra::ModelEvaluatorBase::OutArgs<Scalar>
SecondOrderResidualModelEvaluator<Scalar>::createOutArgsImpl() const {
#ifdef VERBOSE_DEBUG_OUTPUT
  *out_ << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  typedef Thyra::ModelEvaluatorBase MEB;

  MEB::OutArgsSetup<Scalar> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.set_Np_Ng(transientModel_->Np(), 0);
  outArgs.setSupports(MEB::OUT_ARG_f);
  outArgs.setSupports(MEB::OUT_ARG_W_op);

  return outArgs;
}

template <typename Scalar>
void
SecondOrderResidualModelEvaluator<Scalar>::evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
    const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs) const {
#ifdef VERBOSE_DEBUG_OUTPUT
  *out_ << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  typedef Thyra::ModelEvaluatorBase MEB;
  using Teuchos::RCP;

  // Setup initial condition
  // Create and populate inArgs
  MEB::InArgs<Scalar> transientInArgs = transientModel_->createInArgs();

  switch (schemeType_) {
    case NEWMARK_IMPLICIT:

      RCP<Thyra::VectorBase<Scalar> const>
      d = inArgs.get_x();

      RCP<Thyra::VectorBase<Scalar>>
      v = Thyra::createMember(inArgs.get_x()->space());

      RCP<Thyra::VectorBase<Scalar>>
      a = Thyra::createMember(inArgs.get_x()->space());

      // compute acceleration
      // a_{n+1} = (d_{n+1} - d_pred) / dt / dt / beta
      Scalar const
      c = 1.0 / beta_ / delta_t_ / delta_t_;

      Thyra::V_StVpStV(Teuchos::ptrFromRef(*a), c, *d, -c, *d_pred_);

      // compute velocity
      // v_{n+1} = v_pred + \gamma dt a_{n+1}
      Thyra::V_StVpStV(
          Teuchos::ptrFromRef(*v), 1.0, *v_pred_, delta_t_ * gamma_, *a);

      transientInArgs.set_x(d);
      transientInArgs.set_x_dot(v);
      transientInArgs.set_x_dot_dot(a);

      transientInArgs.set_W_x_dot_dot_coeff(c);               // da/dd
      transientInArgs.set_alpha(gamma_ / delta_t_ / beta_);   // dv/dd
      transientInArgs.set_beta(1.0);                          // dd/dd

      break;
  }

  transientInArgs.set_t(t_);
  for (int i = 0; i < transientModel_->Np(); ++i) {
    if (inArgs.get_p(i) != Teuchos::null)
      transientInArgs.set_p(i, inArgs.get_p(i));
  }

  // Setup output condition
  // Create and populate outArgs
  MEB::OutArgs<Scalar> transientOutArgs = transientModel_->createOutArgs();
  transientOutArgs.set_f(outArgs.get_f());
  transientOutArgs.set_W_op(outArgs.get_W_op());

  // build residual and jacobian
  transientModel_->evalModel(transientInArgs, transientOutArgs);
}

}  // namespace Tempus

#endif  // Tempus_SecondOrderResidualModelEvaluator_impl_hpp
